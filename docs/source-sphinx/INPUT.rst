Developing 
============
In order to develop within the code, the reader is encouraged to 
check the `Modules details <_static/doxy/namespaces.html>`_. 

Input file choices
==================

In this section we will describe the input file keywords. 
Every valid keyword will use "camel" syntax and will have 
an ``=`` sign right next to it. For example, the following 
is a valid keyword syntax ``JobName= MyJob``. Comments need 
to have a ``#`` (hash) sign right next to the phrase we want 
to comment. Example comment could be something like:
``#My comment``.  

`JobName=`
***********
This variable will indicate the name of the job we are running. 
It is just a tag to distinguish different outputs. 
As we mentioned before and example use should be: ``JobName= MyJob``

`Verbose=`
*************
Controls the verbosity level of the output. If set to ``False`` no 
output is printed out. If set to ``True``, basic messages of 
the current execution point of the code will be printed. 

