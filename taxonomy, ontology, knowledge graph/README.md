This creates an ontology file from a text input.

You'll need the pre.txt which has template information and your input .txt file with concepts.

Refer to Person.txt for the sample input txt file.

Then just edit create_owl.sh and run it.

<!-- Michael -->
so you need pre.txt which contains template information, a template meaning an already pre-built concept of something to reduce
effort of maybe a certain task

this pre.txt maybe takes in our input .txt file which for for this directory is the Person.txt file as a sample

then just edit the create_owl.sh which is also included in this directory which is a shell file
then run the create_owl.sh

also where does this pre.txt come from?
where does Person.txt come from

this create_owl.sh file which contains the command java -jar ./lib/OWLDump-0.0.1-SNAPSHOT.jar ./Legal.txt ./pre.txt Legal-0.1 ./output/
I believe this command which has the ff
-jar is a flag to read a .jar file which in this case is ./lib/OWLDump-0.0.1-SNAPSHOT.jar
-./Legal.txt is a second argument
-./pre.txt is a third argument
-Legal-0.1 is a fourth argument is inside the ./output/ directory which may be the name that will be given to the .owl file
created by this command resulting in Legal-0.1.owl which is a file in this ./output/ directory
-and ./output/ is a fifth argument which is also in this directory which means it may be that this directory was created by this command

java -jar OWLDump-0.0.1-SNAPSHOT.jar
OwlDump needs 4 args InputFileName, PreTextFileName, OntologyName, OutputDir

ah so the command java -jar ./lib/OWLDump-0.0.1-SNAPSHOT.jar ./Legal.txt ./pre.txt Legal-0.1 ./output/ can be generalized into:
java -jar ./lib/OWLDump-0.0.1-SNAPSHOT.jar <input file name>.txt <pre text file name>.txt <ontology name which will be added an .owl extension later> <output directory to place the .owl file>
