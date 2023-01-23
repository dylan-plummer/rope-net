import os
import uuid
import torch
import flask
import bcrypt
import dash
import numpy as np
import dash_uploader as du
import dash_bootstrap_components as dbc

from flask_login import login_user, LoginManager, UserMixin, logout_user, current_user
from dash import DiskcacheManager, CeleryManager, html, dcc
from dash.dependencies import Output, Input, State

from model import RepNet
from inference import eval_full_video
from utils import remove_directory_containing_file, base64_encode_video, progress_load_model, progress_inference


# https://dash.plotly.com/background-callbacks
if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

# Exposing the Flask Server to enable configuring it for logging in
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server,
                title='RopeNet',
                update_title='Loading...',
                external_stylesheets=[dbc.themes.DARKLY], background_callback_manager=background_callback_manager)
app.config.suppress_callback_exceptions=True               
server = app.server
UPLOAD_FOLDER_ROOT = "uploaded_videos"
TMP_IMG_FOLDER_ROOT = "tmp_imgs"
du.configure_upload(app, UPLOAD_FOLDER_ROOT)
os.makedirs(TMP_IMG_FOLDER_ROOT, exist_ok=True)

center_style = {# wrapper div style
                'textAlign': 'center',
                'width': '40ch',
                'padding': '10px',
                'display': 'inline-block'
            }
# Updating the Flask Server configuration with Secret Key to encrypt the user session cookie
server.config.update(SECRET_KEY=os.getenv('ROPENET_SECRET_KEY'))
# Login manager object will be used to login / logout users
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'
# User data model. It has to have at least self.id as a minimum
class User(UserMixin):
    def __init__(self, username, name=""):
        self.id = username
        self.name = name
@ login_manager.user_loader
def load_user(username):
    ''' This function loads the user by user id. Typically this looks up the user from a user database.
        We won't be registering or looking up users in this example, since we'll just login using LDAP server.
        So we'll simply return a User object with the passed in username.
    '''
    return User(username)

# Login screen
login = html.Div([dcc.Location(id='url_login', refresh=True),
                  html.H2('''Please log in to continue:''', id='h1'),
                  html.Br(),
                  dcc.Input(placeholder='Enter your username',
                            type='text', id='uname-box'),
                  html.Br(),
                  dcc.Input(placeholder='Enter your password',
                            type='password', id='pwd-box'),
                  html.Br(),
                  html.Button(children='Login', n_clicks=0,
                              type='submit', id='login-button'),
                  html.Div(children='', id='output-state'),
                  html.Br(),
                  dcc.Link('Home', href='/')])

# Successful login
success = html.Div([html.Div([html.H2('Login successful.'),
                              html.Br(),
                              dcc.Link('Home', href='/')])  # end div
                    ])  # end div

# Failed Login
failed = html.Div([html.Div([html.H2('Log in Failed. Please try again.'),
                             html.Br(),
                             html.Div([login]),
                             dcc.Link('Home', href='/')
                             ])  # end div
                   ])  # end div

# logout
logout = html.Div([html.Div(html.H2('You have been logged out - Please login')),
                   html.Br(),
                   dcc.Link('Home', href='/')
                   ])  # end div


@app.callback(
    Output('url_login', 'pathname'), 
    Output('output-state', 'children'), 
    [Input('login-button', 'n_clicks')], 
    [State('uname-box', 'value'), State('pwd-box', 'value')],
    prevent_initial_call=True,
    suppress_callback_exceptions=True)
def login_button_click(n_clicks, username, password):
   print(n_clicks, username, password)
   if n_clicks > 0:
      usernames = list(np.loadtxt("user_whitelist.txt", dtype=str))
      print(usernames)
      with open("beta_password", "rb") as f:
         hash_ref = f.read()
      print(usernames)
      pwd_hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
      pwd_check = bcrypt.checkpw(str.encode(password), hash_ref)
      print(pwd_hash, hash_ref, pwd_check)
      if username.lower() in usernames and pwd_check:
         user = User(username)
         login_user(user)
         print(user)
         return '/success', ''
      else:
         return '/login', 'Incorrect username or password'

@app.callback(Output('user-status-div', 'children'), 
              Output('login-status', 'data'), 
              [Input('url', 'pathname')],
              suppress_callback_exceptions=True)
def login_status(url):
    ''' callback to display login/logout link in the header '''
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated \
            and url != '/logout':  # If the URL is /logout, then the user is about to be logged out anyways
        return dcc.Link('logout', href='/logout'), current_user.get_id()
    else:
        return dcc.Link('login', href='/login'), 'loggedout'


def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=1000,  # Mb
        filetypes=['mp4', 'mov'],
        upload_id=uuid.uuid1(),  # Unique session id
    )

base_video_div = html.Video(controls = True,
                              children=html.Source(type='video/mp4'),
                                          id = 'result-video',
                                          autoPlay=False,
                                          style={'display': 'none'})
                                         
base_results_div = html.Div(id='results-container',
                     children=[html.Div(id='results-output', children=[base_video_div]),
                               html.Div(id='progress-container', 
                                        children=[html.P(id='count-pred', children=""),
                                                  html.Button('Submit', id='submit-button', n_clicks=0, disabled=False, style={'display': 'inline-block'}),
                                                  html.P(id="progress-log"),
                                                  html.Progress(id="progress-bar", style={'display': 'none'}),
                                                  html.Br(),
                                                  html.Button('Cancel', id='cancel-button', n_clicks=0, disabled=True, style={'display': 'inline-block'})],
                                        style=center_style),
                               ],
                     style={'textAlign': 'center'})

def get_app_layout():
   main_div = html.Div(
      children=[
            html.Div([get_upload_component(id='dash-uploader'),
                     html.Br(),
                     html.A(html.Button('Reset'), href='/'),
                     html.Br(),
                     html.Div(id='callback-output',
                              children=[html.P(id='video-id', children=[""], style={'display': 'none'})])
               ],
               style=center_style),
            base_results_div
      ],
      style={'textAlign': 'center'}
   )
   output = [dcc.Location(id='url', refresh=False),
            dcc.Location(id='redirect', refresh=True),
            dcc.Store(id='login-status', storage_type='session'),
            html.Div(id='user-status-div'),
            html.Br(),
            html.Hr(),
            html.Br(),
            html.Div([
               html.H3('RopeNet:'),
               dcc.Markdown('''
                  **AI Counting for Competitive Jump Rope**

                  Created by Dylan Plummer in collaboration with the [American Jump Rope Federation](https://www.amjrf.com/)
               ''')], style={'textAlign': 'center'})
            ]
   try:
      if current_user.is_authenticated:
         output.append(main_div)
         return html.Div(id='page-content', children=output)
      else:
         output.append(login)
         return html.Div(id='page-content', children=output, style={'textAlign': 'center'})
   except AttributeError as e:
      print(e)
      output.append(login)
      return html.Div(id='page-content', children=output, style={'textAlign': 'center'})


app.layout = get_app_layout
# get_app_layout is a function
# This way we can use unique session id's as upload_id's

@app.callback(Output('redirect', 'pathname'),
              [Input('url', 'pathname')],
              suppress_callback_exceptions=True)
def display_page(pathname):
   ''' callback to determine layout to return '''
   # We need to determine two things for everytime the user navigates:
   # Can they access this page? If so, we just return the view
   # Otherwise, if they need to be authenticated first, we need to redirect them to the login page
   # So we have two outputs, the first is which view we'll return
   # The second one is a redirection to another page is needed
   # In most cases, we won't need to redirect. Instead of having to return two variables everytime in the if statement
   # We setup the defaults at the beginning, with redirect to dash.no_update; which simply means, just keep the requested url
   url = dash.no_update
   if pathname == '/success':
      if current_user.is_authenticated:
         url = '/'
   elif pathname == '/logout':
      if current_user.is_authenticated:
         logout_user()
      else:
         url = '/login'
   else:
      url = '/'
   # You could also return a 404 "URL not found" page here
   return url


# upload callback
@du.callback(
    Output("callback-output", "children"),
    id="dash-uploader"
)
def callback_on_completion(status: du.UploadStatus):
   video_files = status.uploaded_files
   for video_file in video_files:
      print(video_file)
   output_children = []
   output_children.append(html.P(id='video-id', children=[str(video_file)], style={'display': 'none'}))
   return output_children


# submit video for inference callback
@app.callback(
    Output("progress-bar", "style"),
    Output("progress-log", "children"),
    Input("submit-button", "n_clicks"),
    suppress_callback_exceptions=True
)
def init_progress_bar(n_clicks):
   if n_clicks > 0:
      return {'display': 'inline-block'}, "Getting things set up..."
   else:
      return {'display': 'none'}, ""


# background callback for actual inference
@dash.callback(
    output=Output("results-output", "children"),
    inputs=[Input("submit-button", "n_clicks"), State("video-id", "children"), State("results-output", "children")],
    background=True,
    running=[
        (Output("submit-button", "disabled"), True, False),
        (Output("cancel-button", "disabled"), False, True),
    ],
    cancel=Input("cancel-button", "n_clicks"),
    progress=[Output("progress-bar", "value"), 
              Output("progress-bar", "max"), 
              Output("progress-bar", "style"),
              Output("progress-log", "children"), 
              Output("submit-button", "disabled"),
              Output("cancel-button", "disabled"),
              Output("count-pred", "children")],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def update_progress(set_progress, n_clicks, video_files, current_results):
   print('Callback', n_clicks, video_files)
   if video_files[0] == "":
      return current_results
   elif n_clicks == 0: 
      set_progress((None, None, {'display': 'none'}, "Click submit to analyze your video", True, False, ""))
      return current_results
   else:
      set_progress(progress_load_model())
      video_file = video_files[0]
      use_cuda = torch.cuda.is_available()
      # if not use_cuda:
      #    set_progress((None, None, {'display': 'none'}, "", False, True, "Wait, someone is already using the GPU..."))
      #    return current_results
      # else:
      device = torch.device("cuda" if use_cuda else "cpu")
      print(device)
      model = RepNet(64, backbone='keypoint', backbone_scale='0', trainable_backbone=False, img_size=228)
      model = model.to(device)
      print("loading checkpoint")
      if not use_cuda:
         checkpoint = torch.load("models/ropenet_224.pt", map_location=torch.device('cpu'))
      else:
         checkpoint = torch.load("models/ropenet_224.pt")
      model.load_state_dict(checkpoint['state_dict'], strict=True)
      count_msg = eval_full_video(test_video=str(video_file), model=model, device=device, img_size=228, both_feet=True, 
                     animate=True, out_dir=os.path.join(TMP_IMG_FOLDER_ROOT, os.path.dirname(video_file)), progress_func=set_progress, html_gen_func=progress_inference)
      
      webm_file = os.path.join(TMP_IMG_FOLDER_ROOT, os.path.dirname(video_file), 'anim_0.webm')
      video_code = base64_encode_video(webm_file)
      output_children = html.Video(controls = True,
                                       id = 'result-video',
                                       children=[html.Source(type='video/webm', 
                                                            src=video_code),
                                                "Your browser may not support HTML5 video. Sorry about that."],
                                       key=video_file,
                                       autoPlay=False)
      remove_directory_containing_file(video_file)
      set_progress((None, None, {'display': 'none'}, "Reset to count another video", True, True, count_msg))
      return output_children


if __name__ == '__main__':
   #app.run_server(debug=True, port=8080)
   app.run_server(host="0.0.0.0", debug=False, port=8080)