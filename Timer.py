#If book is a list descrbibing the limit order book and order is an element in the list describing the order
#In this programm I create a list and delete an element of the list after an amount of time

import threading

def cancel_order(book,order):
    for i in range(0,len(book)-1):
        if book[i]==order:
            book.remove(order)
        else:
            pass
    return


LOB=['buy20','sell10','sell50','buy100','sell40'];
order3='sell50';
timer1=threading.Timer(5,cancel_order,[LOB,order3])
timer2=threading.Timer(6,print,['neu= ',LOB])

print('alt= ',LOB)
cancel_order(LOB,order3)
timer1.start()
timer2.start()