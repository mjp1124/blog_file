---
title: 报告邮件自动化发送
categories: email
date: 2017-05-17
tags: 
  - send_email
  - jinja2
---

# 确定要发送的内容
## 将table1,table2作为发送内容


```python
from pandas import DataFrame
import pandas as pd

table1 = DataFrame({'A':[1,2,3,],'B':[2,3,4]})
table2 = DataFrame({'C':[1,3,3,],'D':[2,3,4]})

```
{% asset_img table1.png table1 %} 
{% asset_img table2.png table2 %} 


<!--more-->

# 将table转化成HTML格式
## 此处用到jinja2模板引擎
    * jinja2中有一个核心对象: template Environment(模板环境), 这个类的实例被用于存储配置信息, 全局对象, 从文件系统或其他位置加载模板, 甚至如果你使用Template的构造器创建一个String类型的模板的时候, 也会自动的创建一个Environment对象.
    *  Environment一般在一个应用中只创建一个, 当系统中有不同的配置需求的时候, 就需要创建多个Environment对象相互支持
    *  创建一个Environment对象: env = Environment(loader = FileSystemLoader(‘templates’))…意思是创建一个template Environment对象, 使用的加载器loader是FileSystemLoader类型, 可以加载的模板是当前工作目录下的templates目录下的模板文件
    * 加载一个模板: template = env.get_template(‘mytemplate.html’), 使用env的模板环境加载名为mytemplate.html的模板文件.
    * 渲染一个模板: template.render(date_str=date_str), 渲染模板template, 传入了模板参数date_str值为date_str


```python
from jinja2 import Template, Environment, FileSystemLoader
templateLoader = FileSystemLoader(searchpath="template/") # 我的模板存在了template文件夹
env = Environment(loader=templateLoader)
```

### header模板定义了，我要发送的邮件的开头内容，内容如下:
* 需要传进去的参数为date_str和author

<h4>DEAR ALL～，此为{{date_str}}，自动发送的邮件，请勿直接回复，如有疑问，请与{{author}}联系，谢谢！ </h4>

### main_body_templet模板定义了我要发送的内容，里面内容很简单，因为每次添加的内容只是一张表，所以如下：

```
{% from "detail.html" import table with context %}

{{table(data)}}

```

### 注意上面的模板中，我调用了detail模板，detail内容如下：里面的内容是我对要发送的table的一些定义
* 此处我定义了even和odd是为了不同行显示不同颜色，也可以在style里面定义

```
{% macro table(data) %}
    <table>
         {% for row in data%}
		{% if (row[0]!='序号') and (row[0]%2==0) %}
		<tr class='odd'>
		{% elif (row[0]!='序号') and (row[0]%2!=0) %}
		<tr class ='even'>
		{% else %}
		<tr class ='tableHeader'>
		{% endif %}
                {% for col in row%}
                	<td>{{col}}</td>
                {% endfor %}
            	</tr>
        {% endfor %}
    </table>
{% endmacro %}

```
## 利用模板


```python
start_template = env.get_template('header.html')
table_template = env.get_template('main_body_templet.html')
date_str = datetime.strftime(datetime.now(), '%Y-%m-%d')
start_statement = start_template.render(date_str=date_str,author='Icey')#传进参数并渲染模板
```

* start_statement内容如下


```python
start_statement
```




    '\n<h4>DEAR ALL～，此为2017-05-17，自动发送的邮件，请勿直接回复，如有疑问，请与Icey联系，谢谢！ </h4>\n'



* 下一步我们要去渲染table的模板，此处我定义了一个函数，可以得到我要发送的html,内容如下：


```python
def get_html_text(input_df):
    header = list(input_df.columns)
    header.insert(0, '序号')
    email_list = []
    email_list.append(header)
    email_list.extend(input_df.to_records())
    html_text = table_template.render(data=email_list)
#     html_text = [s for s in html_text.split('\n') if s.replace('\t', '').replace(' ', '') != '']
    return html_text
```


```python
# 传入table,得到对应的html_text
table1_html,table2_html = get_html_text(table1),get_html_text(table2)
```

* 一般情况下我们在发送的表格前都会加一些备注，以便收件人容易理解
* 将我们添加的内容和输出的内容进行组合


```python
mid1 = '<h2>table1的内容:</h2>'
mid2 = '<h2>table2的内容:</h2>'
```


```python
html_text = start_statement + mid1 + table1_html + mid2 + table2_html
html_text = [s for s in html_text.split('\n') if s.replace('\t', '').replace(' ', '') != ''] # 进行一下整理
html_text
```




    ['<h4>DEAR ALL～，此为2017-05-17，自动发送的邮件，请勿直接回复，如有疑问，请与Icey联系，谢谢！ </h4>',
     '<h2>table1的内容:</h2>',
     '    <table>',
     "\t\t<tr class ='tableHeader'>",
     '                \t<td>序号</td>',
     '                \t<td>A</td>',
     '                \t<td>B</td>',
     '            \t</tr>',
     "\t\t<tr class='odd'>",
     '                \t<td>0</td>',
     '                \t<td>1</td>',
     '                \t<td>2</td>',
     '            \t</tr>',
     "\t\t<tr class ='even'>",
     '                \t<td>1</td>',
     '                \t<td>2</td>',
     '                \t<td>3</td>',
     '            \t</tr>',
     "\t\t<tr class='odd'>",
     '                \t<td>2</td>',
     '                \t<td>3</td>',
     '                \t<td>4</td>',
     '            \t</tr>',
     '    </table>',
     '<h2>table2的内容:</h2>',
     '    <table>',
     "\t\t<tr class ='tableHeader'>",
     '                \t<td>序号</td>',
     '                \t<td>C</td>',
     '                \t<td>D</td>',
     '            \t</tr>',
     "\t\t<tr class='odd'>",
     '                \t<td>0</td>',
     '                \t<td>1</td>',
     '                \t<td>2</td>',
     '            \t</tr>',
     "\t\t<tr class ='even'>",
     '                \t<td>1</td>',
     '                \t<td>3</td>',
     '                \t<td>3</td>',
     '            \t</tr>',
     "\t\t<tr class='odd'>",
     '                \t<td>2</td>',
     '                \t<td>3</td>',
     '                \t<td>4</td>',
     '            \t</tr>',
     '    </table>']



## 定义要发送的内容的格式
* 格式这些可以在网上找，有很多漂亮的


```python
styles = '''
<style>
    table {
        background:#87ceeb;
        color: #333; /* Lighten up font color */
        font-family: Helvetica, Arial, sans-serif; /* Nicer font */
        width: 640px;
        border-collapse:
        collapse; border-spacing: 0;
    }

   td, th { border: 1px solid #CCC; height: 30px; } /* Make cells a bit taller */

   th {
        background: #F3F3F3; /* Light grey background */
        font-weight: bold; /* Make sure they're bold */
    }
   td {
        /*background: #FAFAFA;  Lighter grey background */
        text-align: center; /* Center our text */
    }
   .odd>td { background: #FEFEFE;}
   .even>td { background: #F1F1F1;}

</style>

'''
```

## 将发送内容与格式结合在一起，得到最终发送内容及格式


```python
html_text = transform(styles + '\n'.join(html_text))
```

# 定义一个发送邮件的函数


```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from os.path import basename
from premailer import transform

def send_mail(from_addr, to_addr,pass_word, subject, html_text, styles = '', cc_addr = None, bcc_addr = None,  alternative_plain_text = '', local_images = None, attachments = None):
    cc_addr = cc_addr or []
    bcc_addr = bcc_addr or []
    local_images = local_images or {}
    attachments = attachments or []    
    to_all_addr = to_addr+cc_addr+bcc_addr


    html_text = transform(
        styles+html_text
    )

    # Create the root message and fill in the from, to, and subject headers
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    #msgRoot['From'] = from_addr
    #msgRoot['To'] = ','.join(to_addr)
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    # Encapsulate the plain and HTML versions of the message body in an
    # 'alternative' part, so message agents can decide which they want to display.
    msgText = MIMEText(alternative_plain_text)
    msgAlternative = MIMEMultipart('alternative')
    msgAlternative.attach(msgText)

    # We reference the image in the IMG SRC attribute by the ID we give it below
    msgText = MIMEText(html_text, 'html')
    #msgAlternative.attach(msgText)

    msgRoot.attach(msgText)


    msgRoot.add_header('From',from_addr)
    msgRoot.add_header('To', ",".join(to_addr))
    msgRoot.add_header('Cc', ",".join(cc_addr))
    msgRoot.add_header('Bcc', ",".join(bcc_addr))

    # This example assumes the image is in the current directory
    for cid, path in local_images.items():
        fp = open(path, 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()

        # Define the image's ID as referenced above
        msgImage.add_header('Content-ID', '<{cid}>'.format(cid = cid))
        msgRoot.attach(msgImage)

    for path in attachments or []:
        with open(path, "rb") as file:
            part = MIMEApplication(
                file.read(),
                Name=basename(path)
            )
            part['Content-Disposition'] = 'attachment; filename="{}"'.format(basename(path)) 
            msgRoot.attach(part)    

    # Send the email (this example assumes SMTP authentication is required)
    import smtplib
    smtp = smtplib.SMTP()
    smtp.connect('smtp.exmail.qq.com')
    smtp.login(from_addr, pass_word)
    smtp.sendmail(from_addr, to_all_addr, msgRoot.as_string())
    smtp.quit()
```

# 利用上面定义的发邮件的函数，将相应参数传进去就可以发送邮件啦～


```python
subject = '发送邮件报告测试'
alternative_plain_text = 'alternative text'
send_mail(
    '******@**.com',#发件人
    ['******@**.com'],#发件人
    password,
    subject,
    html_text,
    styles=styles,
    cc_addr=None,
    bcc_addr=None,
    alternative_plain_text='alternative text',
    local_images=None,
    attachments=None)
```

# 有时候想发送的邮件中包含很多图片，这时候，我们可以在send_email的local_image定义图片的地址，此处暂不作详细介绍

{% asset_img email_test.png 结果 %} 
