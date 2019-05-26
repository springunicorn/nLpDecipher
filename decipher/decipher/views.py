from django.shortcuts import render
from . import sentiment, classify

def button(request):
    return render(request,'home.html')

def output(request):
    cls = sentiment.run_script()
    return render(request,'home.html',{'data':cls})
