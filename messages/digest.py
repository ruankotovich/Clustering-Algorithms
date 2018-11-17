import csv
import json
import os

messages = []
store_messages = []
clazzes = []
needToLoad = True

if(not(os.path.exists('.step') and os.path.exists('labeledsets.json'))):
    with open('/home/dmitry/Documents/Huge/all-messages.csv') as infile:
        reader = csv.reader(infile)

        for pos, row in enumerate(reader):
            try:

                data = json.loads(row[5], 'utf-8')
                to = json.loads(row[4], 'utf-8')

                if 'blink' in data:
                    del data['blink']
                if 'clientVersion' in data:
                    del data['clientVersion']
                if 'clientName' in data:
                    del data['clientName']

                if 'commandToBot' in data:
                    pass
                elif type(data) == type({})and 'message' in data:
                    text = data['message']
                    messages += [str(text)]
                    store_messages += [{'business': int(to), 'text': str(text)}]
                elif type(data) == type(u'') and 'message' in data:
                    data = json.loads(data, 'utf-8')
                    text = data['message']
                    messages += [str(text)]
                    store_messages += [{'business': int(to), 'text': str(text)}]

            except ValueError as e:
                pass
            except TypeError as e:
                print(e, type(data) == type(u''))
else:
    try:
        stepfile = open('.step', 'r')
        print('Stepfile has been detected, proceeding...')
        unique_message = json.loads(stepfile.read())
        stepfile.close()
        file = open('labeledset.txt', 'r')
        clazzes = json.loads(file.read())
        file.close()
        needToLoad = False
    except Exception as error:
        print(error)
        pass

if(needToLoad):
    unique_message = list(set(messages))
    unique_message_and_business =  list({v['text']:v for v in store_messages}.values())
    f = open('dump.json', 'w')
    f.write(json.dumps(unique_message_and_business))
    print('Unique messsages:', len(unique_message))

file = open('labeledset.txt', 'w')
closedByStep = False

for i, message in enumerate(unique_message):
    print('[Step {s} of {k}] "{m}"'.format(
        s=i+1, k=len(unique_message), m=message))
    dec = input('Is it relevant? (Yes=y/Y | Step=step)')

    if(dec == 'y' or dec == 'Y'):
        more = True
        exp = []
        clazz = input('What is the expected result?')
        exp += [clazz]
        while more:
            more = False
            next = input('Another one? (No=Empty+Enter)')
            if(len(next) > 0):
                more = True
                exp += [next]

        clazzes += [{'message': message, 'expected': exp}]
    elif dec == 'step':
        print('Saved step, bye :D')
        st = open('.step', 'w')
        st.write(json.dumps(unique_message[i:]))
        st.close()
        closedByStep = True
        break

file.write(json.dumps(clazzes))
file.close()

if(not closedByStep and os.path.exists(".step")):
    os.remove(".step")

print("All done!")
