import bottle
from threading import Thread

on_data = lambda alpha, beta, gamma: None

@bottle.route('/')
def handler():
	return bottle.static_file('index.html', '.')

@bottle.post('/data')
def handler():
	on_data(**bottle.request.json)

def data_handler(f):
	global on_data
	on_data = f
	return f

def go():
	t = Thread(target=lambda: bottle.run(quiet=True))
	t.start()