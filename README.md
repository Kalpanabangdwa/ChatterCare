This is a Mental Health Support Chatbot built with Python, Flask, TensorFlow, NumPy, and Pandas. The chatbot provides basic conversational support related to mental health queries.

Features:

Chatbot built using TensorFlow (ML model) for intent classification.

Flask used as the backend to handle chatbot interaction.

Frontend interface for user chat.

Supports answering FAQs on mental health topics.

Project Structure:
Health_BOT/

app.py : Flask server file

chatbot_model.py : Model training and prediction logic

intents.json : Dataset of intents and responses

static/ : CSS, JS files for frontend

templates/ : HTML templates (chat interface)

README.md : Project documentation

Installation & Setup:

Clone the repository
git clone https://github.com/<your-username>/Health_BOT.git
cd Health_BOT

Install dependencies
Make sure you are using Python 3.11 (recommended).
Install the required libraries:
pip install flask tensorflow numpy pandas

Run the chatbot
python app.py

Access in browser
Open http://127.0.0.1:5000/

Future Enhancements:

   1. Deploy on cloud platforms (Heroku, AWS, or Render).

   2. Add user authentication and chat history.

   3. Integrate with real-time APIs for advanced support.

   4. Enhance NLP capabilities for better accuracy.
