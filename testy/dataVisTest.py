import csv
import unittest
from unittest import mock
import json
import os

def openGameFile(name):
    path = os.path.join(name)
    f = open(path)
    data = json.load(f)
    return data

def uploadData(inputFile, outputFile, nadpisz):
  try:
    jsonFile = openGameFile(inputFile)
    clean_event = []
    for event in jsonFile:
      if(event["type"]["name"]=="Shot"):
        clean_event = []
        #time
        clean_event.append(event["minute"])
        #location
        clean_event.append(event["position"]["name"])
        # shot type
        clean_event.append(event["shot"]["type"]["name"])
        #outcome
        clean_event.append(event["shot"]["outcome"]["name"])
        
        clean_event.append(event["shot"]["technique"]["name"])

        m_players = []
        en_players = []
        keeper = []
        for player in event["shot"]["freeze_frame"]:
            player_p = 0
            if player["teammate"] :
                m_players.append(player["location"])
            elif player["position"] == 'Goalkeeper':
                keeper = player['location']
            else:
                en_players.append(player["location"])
        for i in range(0,10):
            if(len(m_players)>i):
              clean_event.append(m_players[i])
            else:
              clean_event.append("none")

        for i in range(0,9):
            if(len(en_players)> i):
              clean_event.append(en_players[i])
            else:
              clean_event.append("none")
              
        clean_event.append(keeper)

        if(event["shot"]["outcome"]=="Goal"):
          clean_event.append(1)
        else:
         clean_event.append(0)
  
     
  #Saving to a file
        with open(outputFile, 'a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(clean_event)

  except:
     raise Exception("invalid data")



class modelAndVisTest(unittest.TestCase):
    
    def dataCleaningTest(self):
        
        # 1().json - Dane w dobrym Formacie
        # 2().json - błedne dane

        uploadData("(1).json", "test_1", True)

        with self.assertRaises(Exception):
           uploadData("(2).json", "test_2", True)

unittest.main()