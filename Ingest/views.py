from django.shortcuts import render
import random

# Create your views here.
def Home(request):
    x = random.randint(0,10)
    print(x)
    return render(request, 'home.html', {'randNum': x})

def Display(request):
    return render(request, 'display.html')

def fetchCases(request):
    return

def Ingest(request):
    # from jsons, ingest to db
    pass
