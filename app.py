from flask import Flask, request, render_template
from flask_mail import Mail, Message
import numpy as np
import pickle
from feature import FeatureExtraction  # Assuming you have a feature extraction module

app = Flask(__name__)

# Configure Flask-Mail with Gmail SMTP server settings
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sarankumaar3@gmail.com'  # Your Gmail email address
app.config['MAIL_PASSWORD'] = 'abdysyazthmsnayf'  # Your Gmail password
app.config['MAIL_DEFAULT_SENDER'] = ('Phishing Detection App', 'sarankumaar3@gmail.com')  # Default sender

mail = Mail(app)

# Load the pre-trained model
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Extract URL from the form data
            url = request.form["url"]
            
            # Extract email addresses (optional)
            emails = request.form.getlist("email")
            
            # Perform feature extraction
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30) 
            
            # Predict using the pre-trained model
            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            x = y_pro_non_phishing
            num = x * 100
            if 0 <= x and x < 0.50:
                num = 100 - num
            formatted_percentage = "{:.2f}%".format(num)
            if y_pred == 1:
                prediction_label = "Safe"
            else:
                prediction_label = "Unsafe"

            # Send an email with the prediction and URL to each recipient
            send_email(url, prediction_label, formatted_percentage, emails)

            # Render the template with the prediction and URL
            return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url, prediction=prediction_label)

        except Exception as e:
            # Handle any errors gracefully
            error_message = "An error occurred: {}".format(str(e))
            return "An error occurred: {}".format(str(e))

    # Render the index template for GET requests
    return render_template("index.html", xx=-1)

def send_email(url, prediction_label, formatted_percentage, emails):
    try:
        for email in emails:
            msg = Message("URL Prediction", recipients=[email])
            msg.body = f"Predicted {url} is {formatted_percentage} {prediction_label}"
            mail.send(msg)
        print("Emails sent successfully")
    except Exception as e:
        print("Error sending emails:", str(e))

if __name__ == "__main__":
    app.run(debug=True)



