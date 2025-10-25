from django.shortcuts import render
import json
from pathlib import Path
import random

# Create your views here.
def Home(request):
    x = random.randint(0,10)
    #get the absolute path of the json files
    base_dir = Path(__file__).resolve().parent 
    parent_dir = base_dir.parent
    #find all json files in the data folder
    for file in parent_dir.glob('*.json'):
        with file.open('r') as f:
            data = json.load(f)
            # loop over each object inside that list
            for obj in data:
                # print each object
                print(obj)
    print(x)
    print("h")


    return render(request, 'home.html', {'randNum': x})

def Display(request):
    return render(request, 'display.html')

def fetchCases(request):
    return

def Ingest(request):
    # from jsons, ingest to db
    pass
