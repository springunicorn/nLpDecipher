from django.shortcuts import render
from . import sentiment, classify

cls = None

def train(request):
    global cls
    cls = sentiment.run_script()

def button(request):
    return render(request,'home.html')

def output(request):
    data = request.POST.get('inputsentence', False)
    # TODO: use trained cls to classify
    return render(request,'home.html',{'data':data})
