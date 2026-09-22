import paho.mqtt.client as mqtt
import time
import queue
import csv
import os
import math

MQTT_HOME_IP = "10.19.4.111"
flag1 = True
max_num_temp = 24
inbound_msg = queue.Queue()
prediction = None
list_of_predict = []
last_sent_time = time.time()
SEND_INTERVAL = 30 #3600  # 1h



def publish_func(outgoing_msg):
    global client
    client.publish("ha/test2", outgoing_msg)

def receive_func():
    global client
    client.on_connect = on_connect
    client.on_message = on_message

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("ha/test")

def on_message(client, userdata, msg):
    if msg.payload.decode().startswith("CSV_DATA:"):
        return
    inbound_msg.put(msg.payload.decode())
    print("RECEIVED:", msg.payload.decode())


def read_from_csv(filename):
    if not os.path.exists(filename):
        print("Datoteka ne postoji:", filename)
        return []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    #return rows
    return create_lop(rows)

def write_to_csv(filename, data_row):
    rows = read_from_csv_internal(filename)
    if len(rows) >= max_num_temp:
        rows = rows[1:]
    rows.append(data_row)
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

def read_from_csv_internal(filename):
    if not os.path.exists(filename):
        print("Datoteka ne postoji:", filename)
        return []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    #return rows
    return rows

def curr_predict():
    lop = []
    lop = read_from_csv("temp2.csv")
    message = "\n".join([";".join(str(temp) for temp in lop)])
    return message
    

def create_lop(rows):
    #global list_of_predict
    #list_of_predict = []
    lop_internal = []
    for row in rows:        
        if row:
            if row[0]:
                broj = float(row[0])
                lop_internal.append(broj)
    return lop_internal

#racunam na 12 podataka
#dobijem median ili srednju vrijednost
#ako su neki članovi za 2 ili 3 veci/manji od te vrijednosti, imamo anomaliju
def anomaly(lop):
    #lop = create_lop()
    avedian = median_temp(lop)
    if avedian == None:
        return "ERROR"
    else:
        return compare(lop, avedian)
    #result = compare(lop, avedian)
    #if result == False:
    #    return False
    #else:
    #    return True

def compare(lop, avedian):
    anomaly_exists = False
    for i in range(0, len(lop)):
        if abs(float(lop[i])-float(avedian)) > 2:
            anomaly_exists = True
    return anomaly_exists

def median_temp(list_of_predict):
    if not list_of_predict:
        return None
    else:
        median = 0.0
        lop_sort = sorted(list_of_predict)
        if len(list_of_predict)%2 == 1:
            median = float(lop_sort[(len(lop_sort) // 2)])
        else:
            median = (lop_sort[len(lop_sort)//2 - 1] + lop_sort[len(lop_sort)//2]) / 2
        return median

def average_temp(list_of_predict):
    if not list_of_predict:
        return None
    else:
        average = 0.0
        for i in range(0, len(list_of_predict)):
            average+=float(list_of_predict[i])
        average = average / float(len(list_of_predict))
        return average

def ensure_connection(max_attempts=3):
    global client
    attempts = 0
    while not client.is_connected():
        attempts += 1
        print(f"Pokušaj ponovnog spajanja: {attempts}/{max_attempts}")
        try:
            client.reconnect()
            print("Ponovno spojen na MQTT broker.")
            return True
        except Exception as e:
            print("Greška pri reconnectu:", e)
            print("Čekam 5 sekundi.")
            time.sleep(5)
        if attempts >= max_attempts:
            print("Neuspješno spajanje nakon 3 pokušaja. Prekidam program.")
            return False
    return True



#MAIN
client = mqtt.Client()

result = client.connect(MQTT_HOME_IP, 1883, 60)
if result != 0:
    print("Ne mogu se spojiti na MQTT Broker.")
    exit(1)
    

receive_func()
client.loop_start()
time.sleep(1)
while flag1:

    if not ensure_connection():
        break

    try:
        message = inbound_msg.get(timeout=5)
    except queue.Empty:
        print("Nema poruke u zadnjih 5 sekundi")
        message = None

    if message:
        if message == "KILL_CLIENT":
            break
        elif message == "":
            print("EMPTY MESSAGE")
        elif message == "SEND_CURR_PREDICTION":
            outgoing_msg = curr_predict()
            publish_func(outgoing_msg)
        else:
            write_to_csv("temp1.csv", [time.strftime("%Y-%m-%d %H:%M:%S"), message])
            print("Poruka spremljena u temp1.csv")

    #provjera tajmera
    if time.time() - last_sent_time >= SEND_INTERVAL:
        outgoing_msg = None
        lop = []
        lop = read_from_csv("temp2.csv")
        status = anomaly(lop)
        if status == True:
            outgoing_msg = "Postoji anomalija"
        elif status == "ERROR":
            outgoing_msg = "ERROR: Incorrect or Missing Data - Can Not Compute"
        else:
            #outgoing_msg = "\n".join([",".join(row) for row in read_from_csv("temp1.csv")])
            outgoing_msg = curr_predict()
        publish_func("CSV_DATA: " + outgoing_msg)
        print("Poslani podaci nakon 1h")
        last_sent_time = time.time()

client.loop_stop()
client.disconnect()
