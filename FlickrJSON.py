# Python program to convert text
# file to JSON
import json

class FlickrJSON:

    def BuildJson(self, Input):
        Input_File_Name = Input
        lists = []
        dict1 = {}
        fields =['file_name', 'image_id', 'caption','captionid']
        index_counter = 0
        # creating dictionary
        with open(Input_File_Name) as fh:
            for line in fh:
                dict1 = {}
                dict1[fields[0]] = line.split('\t', 2)[0].split('#')[0]
                dict1[fields[1]] = index_counter
                dict1[fields[2]] = line.split('\t', 2)[1].replace(" .", ".").replace("\n","")
                dict1[fields[3]] = line.split('\t', 2)[0].split('#')[1]
                index_counter = index_counter + 1
                # command, description = line.strip().split(None, 1)
                # print(dict1)
                lists.append(dict1)
                # dict1[command] = description.strip()
        jsonString = json.dumps({'annotations': lists})
        return jsonString