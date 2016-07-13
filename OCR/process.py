from PIL import Image

f_out = open('data.txt', 'w')

for digit in range(10):
    f_out.write('{0}\n'.format(digit))
    for i in range(100):
        filename = 'data/{0}_{1}.png'.format(digit, i)
        try:
            img = Image.open(filename)
            print 'Opening file "{0}"'.format(filename)
            pix = img.load()
            outputArr = []
            for r in range(img.size[1]):
                for c in range(img.size[0]):
                    val = (255.0 - pix[c, r][0]) / 255.0
                    outputArr.append('{0:.15f}'.format(val))
            f_out.write('{0}\n'.format(','.join(outputArr)))
            
        except IOError:
            break

f_out.close()
