import orange_chatbot as ochat
import socket
from socket_helper import *
import socketserver

class MyTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.orange_chatbot_handler()
    
    def orange_chatbot_handler(self):
        # self.request is the TCP socket connected to the client
        conn = self.request
        address = self.client_address[0]
        print(f"chatbot: connection from: {str(address)}")
        send_msg_real(conn, chatbot.get_intro_line())
        while True:
            # receive input str. 
            inp = get_response_real(conn)
            if not inp:
                # if input is not received, continue and wait for next connection
                print("chatbot: connection closed")
                break
            print(f"chatbot: from connected user: {inp}")
            if inp == ".log":
                send_msg_real(conn, chatbot.get_log())
            elif inp == ".reset":
                chatbot.init_chat_log()
                send_msg_real(conn, "chatbot reset")
            else:
                answer = chatbot.get_response(inp)
                send_msg_real(conn, answer)  # send answer to the client
                if inp.lower() != "goodbye":
                    chatbot.add_to_chat_log(inp, answer)

if __name__ == '__main__':
    # get the hostname
    HOST = socket.gethostname()
    PORT = 5124  # initiate port no above 1024
    print("chatbot: starting chatbot...")

    bot = ochat.OrangeChatbot()
    chatbot = bot.start()

    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        print("chatbot: waiting for connection.")
        server.serve_forever()