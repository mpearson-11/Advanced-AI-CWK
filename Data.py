#   Data Controller
#   @Copywright Max Pearson
#   Student ID: B123103
            
#xlsx conversion
from tabulate import tabulate
import os
import math

def showData(Data):
    print tabulate(Data.document, headers="keys",tablefmt="grid")

def showLegend(Data):
    os.system('clear')
    for i,j in Data.tagLine.items():
        print "{0} = {1}".format(i,j)

class Data:
    def __init__(self):
        self.document=[]
        self.tagLine={}
        self.populate()
        
    def populate(self):
        doc=self.xlsx('CWData.xlsx')

        for i in range(1,10):
            self.tagLine[doc[i]["L"]]=doc[i]["M"]
            
        for i in range(1,len(doc)):
            newObject={}
            newObject["AREA"]=float(doc[i]["A"])
            newObject["BFIHOST"]=float(doc[i]["B"])
            newObject["FARL"]=float(doc[i]["C"])
            newObject["FPEXT"]=float(doc[i]["D"])
            newObject["LDP"]=float(doc[i]["E"])
            newObject["PROPWET"]=float(doc[i]["F"])
            newObject["RMED-1D"]=float(doc[i]["G"])
            newObject["SAAR"]=float(doc[i]["H"])
            newObject["Index flood"]=float(doc[i]["I"])
            self.document.append(newObject)
            

    def xlsx(self,fname):
        #http://stackoverflow.com/questions/4371163/reading-xlsx-files-using-python
        import zipfile
        from xml.etree.ElementTree import iterparse
        z = zipfile.ZipFile(fname)
        strings = [el.text for e, el in iterparse(z.open('xl/sharedStrings.xml')) if el.tag.endswith('}t')]
        rows = []
        row = {}
        value = ''
        for e, el in iterparse(z.open('xl/worksheets/sheet1.xml')):
            if el.tag.endswith('}v'):
                value = el.text
            if el.tag.endswith('}c'): 
                if el.attrib.get('t') == 's':
                    value = strings[int(value)]
                letter = el.attrib['r']
                while letter[-1].isdigit():
                    letter = letter[:-1]
                row[letter] = value
                value = ''
            if el.tag.endswith('}row'):
                rows.append(row)
                row = {}
        return rows

    def getBy(self,tag):
        returnArray=[]
        for newObject in self.document:
            returnArray.append(newObject[tag])
        return returnArray
