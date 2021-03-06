import smtplib
email_user = "mk190@students.kiron.ngo"
email_send = "mk190@students.kiron.ngo"
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(email_user, 'passeport')
message = 'first email from pyton'
server.sendmail(email_user, email_send, message)
server.quit()
