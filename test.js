const fs = require('fs');

fs.readFile("./commit-data.json", "utf8", (error, json_string) => {
    if(error){
        console.log("file reading failed");
        return;
    }
    
    try{
        console.log(json_string)
        
        // define output string
        let output = "";

        // returns a list of all commits
        const commit_data = JSON.parse(json_string);

        // parse each commit and extract message
        for(let commit of commit_data){
            output += `${commit["commit"]["message"]}\n`;
        }
        console.log(output);
        
        fs.writeFile("./parsed-commit-data.txt", output, (error) => {
            if(error){
                console.log("error occured while writing file");
            }else{
                console.log("writing successful");
            }
        })

    }catch(error){
        console.log(`error ${error} occurred`);
    }
});
        

          