import smtplib, os

host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
port = int(os.getenv('SMTP_PORT', 587))
print("Testing connect to", host, port)

try:
    s = smtplib.SMTP(host, port, timeout=10)
    s.ehlo()
    print("Connected, server greeted OK")
    try:
        s.starttls()
        print("STARTTLS succeeded")
    except Exception as e:
        print("STARTTLS failed:", e)
    s.quit()
    print("Connection + TLS OK")
except Exception as e:
    print("Connection failed:", type(e).__name__, e)
