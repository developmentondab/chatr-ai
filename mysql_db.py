import mysql.connector
from datetime import datetime

#mydb = mysql.connector.connect(
#  host="119.18.49.15",
#  user="hrqwqwmy_doc_chat",
#  password="B6l1T0w$c=KY",
#  database="hrqwqwmy_doc_chat"
#)
def connect_to_database():
  return mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="AllowTSL119",
    database="doc_chat"
  )
# print(mydb)
def add_tokens(user_id, access_token, refresh_token):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "INSERT INTO user_tokens (user_id, access_token, refresh_token) VALUES (%s, %s, %s)"
  val = (user_id, access_token, refresh_token)
  mycursor.execute(sql, val)
  mydb.commit()
  row_id = mycursor.lastrowid
  mycursor.close()
  mydb.close()
  return row_id

def check_token(token):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "select * from  user_tokens where (access_token = %s or refresh_token = %s) and user_tokens.revoke = %s"
  val = (token, token, '1')
  mycursor.execute(sql, val)

  rows = mycursor.fetchall() 
  result = [
        {"user_id": row[1], 
        "access_token": row[2], 
        "refresh_token": row[3], 
        "id": row[0]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  mydb.close()
  return result

def revoke_token(user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "UPDATE user_tokens SET user_tokens.revoke = %s WHERE user_id = %s"
  val = ('1', user_id)
  mycursor.execute(sql, val)
  mydb.commit()
  
  result = mycursor.rowcount
  if(mycursor.rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def check_login(data):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "select * from  users where email = %s and password = %s"
  val = (data['username'], data['password'])
  mycursor.execute(sql, val)

  rows = mycursor.fetchall() 
  result = [
        {"username": row[1], 
        "email": row[2], 
        "first_name": row[4], 
        "last_name": row[5], 
        "status": row[6], 
        "user_id": row[0]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  mydb.close()
  return result

def check_email(email):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM users WHERE email = %s"
  adr = (email, )
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  result = [
        {"username": row[1], 
        "email": row[2], 
        "first_name": row[4], 
        "last_name": row[5], 
        "status": row[6], 
        "user_id": row[0],
        "password": row[3]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  mydb.close()
  return result

def check_googleid(google_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM users WHERE google_id = %s"
  adr = (google_id, )
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  result = [
        {"username": row[1], 
        "email": row[2], 
        "first_name": row[4], 
        "last_name": row[5], 
        "status": row[6], 
        "user_id": row[0],
        "password": row[3]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  return result

def add_user(data):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  if 'google_id' in data:
    google_id = data['google_id']
  else:
    google_id = ''

  if 'cookie_id' in data:
    cookie_id = data['cookie_id']
  else:
    cookie_id = ''
  sql = "INSERT INTO users (username, email, password, first_name, last_name, google_id, cookie_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
  val = (data['username'], data['email'], data['password'], data['first_name'], data['last_name'], google_id, cookie_id)
  mycursor.execute(sql, val)
  mydb.commit()
  row_id = mycursor.lastrowid
  mycursor.close()
  mydb.close()
  return row_id

def get_user(user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM users WHERE id = %s"
  adr = (user_id, )
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  result = [
        {"username": row[1], 
        "email": row[2], 
        "first_name": row[4], 
        "last_name": row[5], 
        "status": row[6], 
        "user_id": row[0], 
        "cookie_id": row[8]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  mydb.close()
  return result

def add_collection(data):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "INSERT INTO collections (collection, user_id, kb_name, data_type, file_paths, file_names) VALUES (%s, %s, %s, %s, %s, %s)"
  val = (data['collection_name'], data['user_id'], data['kb_name'], data['data_type'], str(data['file_paths']), str(data['file_names']))
  mycursor.execute(sql, val)
  mydb.commit()
  row_id = mycursor.lastrowid
  mycursor.close()
  mydb.close()
  return row_id

def get_collection(user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM collections WHERE user_id = %s order by id desc"
  adr = (user_id, )
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  
  result = {
        row[2]:{"user_id": row[1], 
        "collection": row[2], 
        "kb_name": row[3], 
        "data_type": row[4], 
        "file_paths": eval(row[5]), 
        "file_names": eval(row[6])
        } for row in rows}
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def get_collection_bykey(user_id, collection):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM collections WHERE user_id = %s and collection = %s order by id desc"
  adr = (user_id, collection)
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  result = [
        {"user_id": row[1], 
        "collection": row[2], 
        "kb_name": row[3], 
        "data_type": row[4], 
        "file_paths": eval(row[5]), 
        "file_names": eval(row[6])
        } for row in rows]
   
  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  mydb.close()
  return result

def remove_collection(user_id, collection):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "DELETE FROM collections WHERE user_id = %s and collection = %s"
  adr = (user_id, collection)
  mycursor.execute(sql, adr)
  mydb.commit()

  result = mycursor.rowcount
  if(result <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def update_user(data, user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "UPDATE users SET username = %s, email = %s, first_name = %s, last_name = %s WHERE id = %s"
  val = (data['username'], data['email'], data['first_name'], data['last_name'], user_id)

  mycursor.execute(sql, val)
  mydb.commit()
  
  result = mycursor.rowcount
  if(mycursor.rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def get_share_data(share_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM shared_chats WHERE share_id = %s"
  val = (share_id,)

  mycursor.execute(sql, val)  
  rows = mycursor.fetchall() 
  
  result = [
        {"share_id": row[1], 
        "shared_user_id": row[2], 
        "shared_content": row[4],  
        "collection": row[3],
        "created_at": row[5], 
        "updated_at": row[6]
        } for row in rows]
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()  
  mydb.close()
  return result

def get_user_shares(user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM shared_chats WHERE shared_user_id = %s"
  val = (user_id,)

  mycursor.execute(sql, val)
  rows = mycursor.fetchall() 
  
  result = [
        {"share_id": row[1], 
        "shared_user_id": row[2], 
        "shared_content": row[4], 
        "collection": row[3],
        "created_at": row[5], 
        "updated_at": row[6]
        } for row in rows]
  
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def add_share_data(share_id, shared_user_id, collection, shared_content):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "INSERT INTO shared_chats (share_id, shared_user_id, collection, shared_content) VALUES (%s, %s, %s, %s)"
  val = (share_id, shared_user_id, collection, shared_content)
  mycursor.execute(sql, val)
  mydb.commit()
  row_id = mycursor.lastrowid
  mycursor.close()
  mydb.close()
  return row_id

def check_share_data(user_id, collection):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM shared_chats WHERE shared_user_id = %s and collection =  %s order by id desc limit 1"
  val = (user_id, collection)
  mycursor.execute(sql, val)  
  rows = mycursor.fetchall() 
  
  result = [
        {"id": row[0],
        "share_id": row[1], 
        "shared_user_id": row[2], 
        "shared_content": row[4],  
        "collection": row[3],
        "created_at": row[5], 
        "updated_at": row[6]
        } for row in rows]
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def update_share_data(id, shared_content):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "UPDATE shared_chats set shared_content = %s WHERE id = %s"
  val = (shared_content, id)

  infoo = mycursor.execute(sql, val)
  mydb.commit()
  
  result = mycursor._rowcount
  if(mycursor.rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def get_subscription(user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM user_subscriptions WHERE user_id = %s and status=1"
  val = (user_id,)

  mycursor.execute(sql, val)  
  rows = mycursor.fetchall() 
  
  result = [
        {"user_id": row[1], 
        "sub_id": row[2], 
        "name": row[3],
        "start_date": row[4],  
        "end_date": row[5], 
        "invoice_url": row[6],
        "status": row[7]
        } for row in rows]
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def add_subscription_data(data, user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "INSERT INTO user_subscriptions (user_id, sub_id, name, start_date, end_date, invoice_url, status) VALUES (%s, %s, %s, %s, %s, %s, %s)"
  val = (user_id, data['sub_id'], data['name'], datetime.fromtimestamp(data['start_date']), datetime.fromtimestamp(data['end_date']), data['invoice_url'], data['status'])
  mycursor.execute(sql, val)
  mydb.commit()
  row_id = mycursor.lastrowid
  mycursor.close()
  mydb.close()
  return row_id

def update_subscription_data(data, user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "UPDATE user_subscriptions set shared_content = %s WHERE user_id = %s and sub_id = %s"
  val = (data, user_id, sub_id)

  mycursor.execute(sql, val)
  mydb.commit()
  
  result = mycursor.rowcount
  if(mycursor.rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def update_subscription_status(status, sub_id, user_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "UPDATE user_subscriptions set status = %s WHERE user_id = %s and sub_id = %s"
  val = (status, user_id, sub_id)

  mycursor.execute(sql, val)
  mydb.commit()
  
  result = mycursor.rowcount
  if(mycursor.rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result

def get_share_user_data(share_id):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM shared_chats WHERE share_id = %s"
  val = (share_id,)

  mycursor.execute(sql, val)  
  rows = mycursor.fetchall() 
  
  result = [
        {"share_id": row[1], 
        "shared_user_id": row[2], 
        "collection": row[3]
        } for row in rows]
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()  
  mydb.close()
  return result

def check_cookieid_email(cookie_id, email):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  if(cookie_id != ""):
    sql = "SELECT * FROM users WHERE cookie_id = %s or email= %s"
    adr = (cookie_id, email)
  else:
    sql = "SELECT * FROM users WHERE email= %s"
    adr = (email,)

  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  result = [
        {"username": row[1], 
        "email": row[2], 
        "first_name": row[4], 
        "last_name": row[5], 
        "status": row[6], 
        "user_id": row[0],
        "password": row[3],
        "cookie_id": row[8]
        } for row in rows]

  if(mycursor._rowcount <= 0): 
    result = False
  else:
    result = result[0]
  mycursor.close()
  return result

def get_collection_byDate(user_id, date):
  mydb = connect_to_database()
  mycursor = mydb.cursor()
  sql = "SELECT * FROM collections WHERE user_id = %s and DATE(created_at) = %s order by id desc"
  adr = (user_id, date)
  mycursor.execute(sql, adr)

  rows = mycursor.fetchall() 
  
  result = {
        row[2]:{"user_id": row[1], 
        "collection": row[2], 
        "kb_name": row[3], 
        "data_type": row[4], 
        "file_paths": eval(row[5]), 
        "file_names": eval(row[6])
        } for row in rows}
 
  if(mycursor._rowcount <= 0): 
    result = False
  mycursor.close()
  mydb.close()
  return result
