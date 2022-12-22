Usage
=====

Installation
------------

To use this API, plese install python3 and the requirement libraries.

.. code-block:: console

   (.venv) $ pip3 install -r requirements.txt



To start the API 
----------------

.. code-block:: console

   (.venv) $ cd api
   (.venv) $ python3 app.p

Endpoints for this API
=====================

Endpoint: search
----------------

To download the files from NASA website you can use the endpoint /search given a topic.

``search(userInputTopic, lang="en"):`` function:

.. py:function:: search(userInputTopic, lang="en")

   Return a link to the path of files as strings.

   :param userInputTopic: user input topic.
   :type kind: str
   :return: The ingredients list.
   :rtype: str


Endpoint: extractInfoFromPaper
------------------------------

To extract text and images from papers you can use the /extractInfoFromPaper endpoint

``extractInfoFromPaper (userInputTopic, lang="en"):`` function:

.. py:function:: extractInfoFromPaper(userInputTopic, lang="en")

   Return a list of link to the path of pdf and text and images as strings.

   :param userInputTopic: user input topic.
   :type kind: str
   :return: The ingredients list.
   :rtype: list[str]

Endpoint: findTopicForPapers
----------------------------

To find the topic of the papers you can use the /findTopicForPapers endpoint

``findTopicForPapers(userInputTopic, lang="en"):`` function:

.. py:function:: findTopicForPapers(userInputTopic, lang="en")

   Return a list of link to the path of pdf and text and images as strings.

   :param userInputTopic: user input topic.
   :type kind: str
   :return: The ingredients list.
   :rtype: list[str]


