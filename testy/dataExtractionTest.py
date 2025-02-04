
import csv
import unittest
from unittest import mock


def GainPossition(n,file):
    ret = ""
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for i in range(1,n):
            ret = next(reader)
    return ret


class modelAndVisTest(unittest.TestCase):
    
    def modelAndVisTest(self):
        
        #Wiersze ze zbioru:
        # Test sprawdzający czy funkcja wyciąga właściwe ustawienie z pliku

        file = "test_dataSet.csv"

        wiersz_1_5 = [113.0, 38.0]

        wiersz_1_2 = "Right Center Forward"

        wiersz_7_3 = "Open Play"

        wiersz_23_4 = "Off T"


        # Pobranie pozycji za pomocą testowanej funkcji

        p_wiersz_1 = GainPossition(1,file)
        
        p_wiersz_7 = GainPossition(7,file)
        
        p_wiersz_23 = GainPossition(23,file)
        
        self.assertEquals(wiersz_1_5,p_wiersz_1[5])
        self.assertEquals(wiersz_1_2,p_wiersz_1[2])
        self.assertEquals(wiersz_7_3,p_wiersz_1[3])
        self.assertEquals(wiersz_23_4,p_wiersz_1[4])

unittest.main()