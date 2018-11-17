const fs = require('fs');
const path = require('path');

(() => {
    let testCollection = [];

    for (let file of fs.readdirSync('./').filter((k) => path.extname(k).toLowerCase() === '.piece')) {
        let originalObject = JSON.parse(fs.readFileSync(file, 'utf-8'));
        testCollection.push({
            business: originalObject.business,
            input: originalObject.text,
            expected: originalObject.entities.map(
                el => el.label
            )
        });
    }
    fs.writeFileSync('../testcollection.json', JSON.stringify(testCollection, null, 2));
})();