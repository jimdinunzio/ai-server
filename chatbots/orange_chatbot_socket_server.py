import orange_chatbot as ochat
import socket
from socket_helper import *

def orange_chatbot_server():
    print("chatbot: starting chatbot...")
    
    bot = ochat.OrangeChatbot()
    chatbot = bot.start()

    # get the hostname
    host = socket.gethostname()
    PORT = 5124  # initiate port no above 1024

    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, PORT))  # bind host address and port together
    server_socket.listen()
    conn = None
    print("chatbot: started")
    while True:
        try:
            print("chatbot: waiting for connection.")
            conn, address = server_socket.accept()  # accept new connection
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
                elif inp == ".restart":
                    send_msg_real(conn, "chatbot restarting")
                    chatbot = None
                    bot.stop()
                    bot = None
                    return
                else:
                    answer, _, long = chatbot.get_response(inp)
                    print(f"\n------------------------------\n{long}\n-----------------------------\n")
                    send_msg_real(conn, answer)  # send answer to the client
                    inp = inp.lower()
                    if "goodbye" not in inp:
                        chatbot.add_to_chat_log(inp, answer)
        except KeyboardInterrupt:
            if conn is not None:
                conn.close()  # close the connection
            break
        except Exception as e:
            if conn is not None:
                conn.close()  # close the connection
            print(f"chatbot: {e}")

if __name__ == '__main__':
    while True:
        orange_chatbot_server()