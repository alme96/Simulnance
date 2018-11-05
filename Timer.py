#If book is a list descrbibing the limit order book and order is an element in the list describing the order
#In this programm I create a list and delete an element of the list after an amount of time

import threading

def cancel_order(book,order):
    book.remove(order)
    return book

book=[1,2,3,4,5,6,7,8,9,10];
order=6;
timer1=threading.Timer(5,cancel_order,[book,order])
timer2=threading.Timer(6,print,['neu= ',book])

print('alt= ',book)
timer1.start()
timer2.start()
