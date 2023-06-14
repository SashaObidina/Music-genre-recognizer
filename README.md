# Music-genre-recognizer
Client-server application using artificial intelligence to recognize the genre of music.

## How to run



### Client-server

Client is an interactive prompt client.

Run server as:
```shell
uvicorn server:app --host 127.0.0.1 --port 8000 --reload 
```
Use `--reload` flag only for debugging.
Then start client as:
```shell
python client_rest.py
```
and run it in interactive shell.
