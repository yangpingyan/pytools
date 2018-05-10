# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:34:44 2017

@author: yangp
"""


from PyPDF2 import PdfFileWriter, PdfFileReader

def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='split pdf file')

    parser.add_argument('--startpage', required=False, action='store',
                        type=float, default=0,          
                        help=('start page of the pdf'))
    
    if pargs is not None:
        return parser.parse_args(pargs)

    return parser.parse_args()



def CreatePdf(input_pdf, start_page, page_num):
    file_input = open(input_pdf, "rb")
    input1 = PdfFileReader(file_input)
    output = PdfFileWriter()
#    file_pages = input1.getNumPages()
#    print("document1.pdf has %d pages." % file_pages)
    for page in range(start_page, start_page+page_num):      
        print(page)
        output.addPage(input1.getPage(page))
    file_output = open("xh{}.pdf".format(start_page), "wb")
    output.write(file_output)    
    file_output.close()
    file_input.close()
    print("One Mission Complete!") 


        
file_pdf = '/Users/zhanghua/Downloads/般若西湖系列2017年度三季度披露报告.pdf'
#file_pdf = 'C:/Users/yangp/iCloudDrive/test/黄龙系列2017年三季度报告.pdf'


file_pages = 104
step_size = 4
page_pos = 0


while page_pos+step_size <= file_pages:
    CreatePdf(file_pdf, page_pos, 4)
    page_pos += step_size

#CreatePdf(input1, page_pos, page_pos+step_size)



print("Final Mission Complete!")    

    
    
    
    

#from PyPDF2 import PdfFileWriter, PdfFileReader
#
#output = PdfFileWriter()
#input1 = PdfFileReader(open("document1.pdf", "rb"))
#
## print how many pages input1 has:
#print "document1.pdf has %d pages." % input1.getNumPages()
#
## add page 1 from input1 to output document, unchanged
#output.addPage(input1.getPage(0))
#
## add page 2 from input1, but rotated clockwise 90 degrees
#output.addPage(input1.getPage(1).rotateClockwise(90))
#
## add page 3 from input1, rotated the other way:
#output.addPage(input1.getPage(2).rotateCounterClockwise(90))
## alt: output.addPage(input1.getPage(2).rotateClockwise(270))
#
## add page 4 from input1, but first add a watermark from another PDF:
#page4 = input1.getPage(3)
#watermark = PdfFileReader(open("watermark.pdf", "rb"))
#page4.mergePage(watermark.getPage(0))
#output.addPage(page4)
#
#
## add page 5 from input1, but crop it to half size:
#page5 = input1.getPage(4)
#page5.mediaBox.upperRight = (
#    page5.mediaBox.getUpperRight_x() / 2,
#    page5.mediaBox.getUpperRight_y() / 2
#)
#output.addPage(page5)
#
## add some Javascript to launch the print window on opening this PDF.
## the password dialog may prevent the print dialog from being shown,
## comment the the encription lines, if that's the case, to try this out
#output.addJS("this.print({bUI:true,bSilent:false,bShrinkToFit:true});")
#
## encrypt your new PDF and add a password
#password = "secret"
#output.encrypt(password)
#
## finally, write "output" to document-output.pdf
#outputStream = file("PyPDF2-output.pdf", "wb")
#output.write(outputStream)
