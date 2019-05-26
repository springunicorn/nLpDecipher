from django.shortcuts import render

def button(request):
    return render(request,'home.html')

def output(request):
    data = request.POST.get('inputsentence', False);
    return render(request,'home.html',{'data':data})
