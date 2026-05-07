from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required


def landing(request):

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")
        else:
            return render(request, "landing.html", {
                "error": "Invalid username or password"
            })

    return render(request, "landing.html")


@login_required
def home(request):
    return render(request, "home.html")


def logout_view(request):
    logout(request)
    return redirect("landing")