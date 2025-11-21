# smtp_send_test.py
import os, smtplib
from email.message import EmailMessage

host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
port = int(os.getenv('SMTP_PORT', 587))
user = os.getenv('SMTP_USER')
pwd  = os.getenv('SMTP_PASS')
fr   = os.getenv('FROM_EMAIL', user)
to   = user  # send to yourself for test

msg = EmailMessage()
msg['From'] = fr
msg['To'] = to
msg['Subject'] = 'SMTP Test from local dev'
msg.set_content('If you receive this, SMTP login and send worked.')

print("Connecting to", host, port, "as", user)
try:
    with smtplib.SMTP(host, port, timeout=15) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        if user and pwd:
            s.login(user, pwd)
        s.send_message(msg)
    print("Send succeeded — check inbox for:", to)
except Exception as e:
    print("Send failed:", type(e).__name__, e)
