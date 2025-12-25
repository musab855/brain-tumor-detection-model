from flask import Flask
from flask_routes import routes

app = Flask(__name__)
app.secret_key = 'my-very-long-random-string-that-is-hard-to-guess'
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)
