from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS, cross_origin
import urllib.request
from ocr_pipeline import process

app = Flask(__name__)
api = Api(app)
CORS(app)

def createImg(imgUrl):
    urllib.request.urlretrieve(imgUrl, "newimage.jpg")
    # pushLatLng()

def bufferProcess():
    return process('./newimage.jpg',1)

class HelloWorld(Resource):
    @cross_origin()
    def get(self):
        lat, lng = bufferProcess();
        # lat = 11.193712907467816 
        # lng = 106.5878921136777
        point = {}
        point['lat'] = lat
        point['lng'] = lng
        print(lat, lng)
        return jsonify(point)

    @cross_origin()
    def post(self):
        res = request.data.decode('utf-8')
        createImg(res)
        return jsonify({'message': 'ok'})

    @cross_origin()
    def options(self):
        pass

api.add_resource(HelloWorld, '/api', endpoint='api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)
