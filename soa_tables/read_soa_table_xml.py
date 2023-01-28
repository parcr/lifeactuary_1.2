__author__ = "PedroCR"

from xml.dom import minidom

http_header = 'https://mort.soa.org/ViewTable.aspx?&TableIdentity='

# todo: correct nan when reading tables from excel with different w's.
class SoaTable:
    def __init__(self, table_name):
        '''
        Reads a previously downloaded life table, from Society of Actuaries, in the xml format and prepares
        all the information to create a life (mortality table).
        :param table_name: The SOA table, in xml format, to be read.
        '''
        self.table_name = table_name
        self.xmldoc = minidom.parse(table_name)
        self.table_id = self.xmldoc.getElementsByTagName('TableIdentity')[0].childNodes[0].data
        self.url = http_header + self.table_id
        self.name = self.xmldoc.getElementsByTagName('TableName')[0].childNodes[0].data
        self.contentType = self.xmldoc.getElementsByTagName('ContentType')[0].childNodes[0].data
        self.tableReference = self.xmldoc.getElementsByTagName('TableReference')[0].childNodes[0].data
        self.ages = self.xmldoc.getElementsByTagName('Y')
        self.min_age = int(self.ages[0].attributes['t'].value)
        self.max_age = self.min_age + len(self.ages) - 1
        self.table_qx = [float(age.childNodes[0].data) for age in self.ages]
        self.table_qx.insert(0, self.min_age)
        # todo: scrap the tables from SOA
