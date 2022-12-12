from flask import Flask, render_template, request, redirect, url_for, session
from search import search
from gettopic import gettopic
from flask import send_from_directory
from getsimilarity import getsimilarity

app = Flask(__name__)
app.secret_key = "#$%#$%^%^BFGBFGBSFGNSGJTNADFHH@#%$%#T#FFWF$^F@$F#$FW"


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/search", methods=["POST", "GET"])
def searchr():
	if request.method == "POST":
		query = request.form["query"]
		results = search(query)
		session["results"] = results
		session["query"] = query
		return redirect(url_for("searchr"))

	return render_template(f"index.html", results=session["results"], query=session["query"])

@app.route("/gettopic", methods=["POST", "GET"])
def gettopicr():
	if request.method == "POST":
		query = request.form["query"]
		results = gettopic(query)
		session["results"] = results
		session["query"] = query
		return render_template("getUserInputArticle.html", results=session["results"], query=session["query"])

	if request.method == "GET":
		return render_template("getUserInputArticle.html")

	return render_template(f"getUserInputArticle.html", results=session["results"], query=session["query"])

@app.route("/getsimilarity", methods=["POST", "GET"])
def getsimilarityr():
	if request.method == "POST":
		query1 = request.form["query1"]
		query2 = request.form["query2"]
		results = getsimilarity(query1, query2)
		session["results"] = results
		return render_template("getArticlesFromUser.html", results=session["results"])

	if request.method == "GET":
		return render_template("getArticlesFromUser.html")

	return render_template(f"getArticlesFromUser.html", results=session["results"], query=session["query"])


@app.route("/lda_html/txt/<string:subpath>")
def lda(subpath):
	return render_template(f"lda_html/txt/{subpath}")


@app.route("/table/<path:path>")
def table(path):
	return send_from_directory('table', path)

@app.route("/lda_model/<path:path>")
def lda_model(path):
	return send_from_directory('lda_model', path)

@app.route("/ldaclimatechange/<path:path>")
def lda_model_climate(path):
	return send_from_directory('ldaclimatechange', path)


if __name__ == '__main__':
	app.run(debug=True)
