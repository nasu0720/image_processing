from flask import Flask

app = Flask(__name__)
app.config.from_object('webapl.config')



import webapl.views