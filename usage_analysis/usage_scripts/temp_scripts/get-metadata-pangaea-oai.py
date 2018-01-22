from sickle import Sickle
#
sickle = Sickle('https://ws.pangaea.de/oai/provider')
records = sickle.ListRecords(metadataPrefix='pan_md')
#with open('response.xml', 'wb') as fp:
    #fp.write(records.next().raw.encode('utf8'))

for record in records:
    header = record.header
    print('id: {}'.format(header.identifier))

    metadata = record.metadata
    if 'title' in metadata:
        print('title: {}'.format(metadata['title']))
