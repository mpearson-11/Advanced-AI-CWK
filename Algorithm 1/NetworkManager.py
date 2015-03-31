from cPickle import *

class NetworkManager:

	def __init__(self,name):
		self.name=name
	
	def saveThis(self,objectN):
		File=open(self.name,"w")
		dump(objectN,File)
		File.close()
	
	def load(self):
		File=open(self.name,"r")
		OBJECT=load(File)
		File.close()
		return OBJECT



