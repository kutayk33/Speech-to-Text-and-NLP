import smtplib
email_user = "email"
email_send = "email"
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(email_user, 'passeport')
message = 'first email from pyton'
server.sendmail(email_user, email_send, message)
server.quit()
