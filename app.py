from io import BytesIO
import base64
from flask import Flask, render_template, Response,request,send_file
import cv2
import predict
app = Flask(__name__)
cap = cv2.VideoCapture(0)  
p=0
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            q1,frame=predict.predict_without_helmet(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
x=0
@app.route('/hello')
def index():
    global p
    p=predict.select_head()
    return render_template('index1.html',x=p)
@app.route('/')
def select():
    global x
    x,y=predict.select_vehicle()
    return render_template('select.html',x=x)
@app.route('/twowheeler',methods=['POST'])
def twowheeler():
    for i in range(x):
        y=request.form.get(str(i))
        if y!=None and y==str(i): 
            _,img=predict.helper(i)  
            print(type(img))
    ret,encoded=cv2.imencode(".jpg",img)
    print(type(encoded))
    image_stream = BytesIO(encoded.tobytes())
    
    # Instead of returning ndarray, use send_file to send the image
    return send_file(image_stream, mimetype='image/jpeg')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
