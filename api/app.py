from flask import Flask, render_template, request, redirect, url_for, session
from search import search
from gettopic import gettopic
from flask import send_from_directory
from getsimilarity import getsimilarity
from extractInfoFromPapers import extractInfoFromPapers
from findTopicForPapers import findTopicForPapers
from findSimilarityForPapers import findSimilarityForPapers
from getImagesFromFile import getImagesFromFile
app = Flask(__name__)
app.secret_key = "#$%#$%^%^BFGBFGBSFGNSGJTNADFHH@#%$%#T#FFWF$^F@$F#$FW"


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/search", methods=["POST", "GET"])
def searchr():
	if request.method == "POST":
		query = request.form["query"]
		results, table_name = search(query)
		session["results"] = results
		session["table_name"] = table_name
		session["query"] = query
		return redirect(url_for("searchr"))

	return render_template(f"index.html", results=session["results"], table_name=session["table_name"], query=session["query"])

@app.route("/extractInfoFromPapers", methods=["POST", "GET"])
def extractInfoFromPapersr():
	if request.method == "POST":
		query = request.form["query"]
		results, table_name = extractInfoFromPapers(query)
		session["results"] = results
		session["table_name"] = table_name
		session["query"] = query
		return render_template("extractInfoFromPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])
	if request.method == "GET":
		return render_template("extractInfoFromPapers.html")


	return render_template(f"extractInfoFromPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])

@app.route("/findTopicForPapers", methods=["POST", "GET"])
def findTopicForPapersr():
	if request.method == "POST":
		query = request.form["query"]
		results, table_name = findTopicForPapers(query)
		session["results"] = results
		session["table_name"] = table_name
		session["query"] = query
		return render_template("findTopicForPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])
		
	if request.method == "GET":
		return render_template("findTopicForPapers.html")


	return render_template(f"extractInfoFromPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])

@app.route("/findSimilarityForPapers", methods=["POST", "GET"])
def findSimilarityForPapersr():
	if request.method == "POST":
		query = request.form["query"]
		results, table_name = findSimilarityForPapers(query)
		session["results"] = results
		session["table_name"] = table_name
		session["query"] = query
		return render_template("findSimilarityForPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])
		
	if request.method == "GET":
		return render_template("findSimilarityForPapers.html")


	return render_template(f"findSimilarityForPapers.html", results=session["results"], table_name=session["table_name"], query=session["query"])


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

@app.route("/getImagesFromFile", methods=["POST", "GET"])
def getImagesFromFiler():
	if request.method == "POST":
		query = request.form["query"]
		results = getImagesFromFile(query)
		session["results"] = results
		return render_template("getImagesFromFile.html", results=session["results"])

	if request.method == "GET":
		return render_template("getImagesFromFile.html")

	return render_template(f"getImagesFromFile.html", results=session["results"], query=session["query"])


@app.route("/lda_html/txt/<string:subpath>")
def lda(subpath):
	return render_template(f"lda_html/txt/{subpath}")


@app.route("/<path:main_path>/<path:path>")
def displayLink(main_path, path):
	return send_from_directory(f'{main_path}', path)


if __name__ == '__main__':
	app.run(debug=True)
