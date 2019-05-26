from django.shortcuts import render

def button(request):
    return render(request,'home.html')

def output(request):
    data = 123
    return render(request,'home.html',{'data':data})
