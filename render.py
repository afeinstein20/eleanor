import os

from jinja2 import Environment, FileSystemLoader

from bs4 import BeautifulSoup
from pygments import highlight
from pygments.lexers import guess_lexer
from pygments.formatters import HtmlFormatter

html_formatter = HtmlFormatter()

# this tells jinja2 to look for templates
# in the templates subdirectory
env = Environment(
    loader = FileSystemLoader('templates'),
)

input_file = 'main.html'
output_file = 'index.html'

# reading the template
template = env.get_template(input_file)
# render the template
# in other words, we replace the template tag
# by the contents of the overfitting file
rendered = template.render()

# replace the pre tags by highlighted code
soup = BeautifulSoup(rendered, 'html.parser')
for pre in soup.find_all('pre'):
    # escaping pres in the jupyter notebook
    # either they're already formatted (input code),
    # or they should remain unformatted (ouput code)
    if pre.parent.name == 'div':
        pclass = pre.parent.get('class')
        if pclass and \
           ('highlight' in pclass or \
            'output_text' in pclass):
            continue
    # highlighting with pygments
    lexer = guess_lexer(pre.string)
    code = highlight(pre.string.rstrip(),
                     lexer, html_formatter)
    # replacing with formatted code
    new_tag = pre.replace_with(code)
    print(new_tag)

# create the final html string.
# formatter None is used to preserve
# the html < and > signs
rendered = soup.prettify(formatter=None)

# write the result to disk in index.html
with open(output_file, 'w') as ofile:
    ofile.write(rendered)
