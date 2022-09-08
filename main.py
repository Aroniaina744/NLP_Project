import tornado.ioloop
import tornado.web
from models import predict
import json

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        msg = self.get_argument("m")
        p = predict(msg)
        self.write(json.dumps(p))

def make_app():
	    return tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
