var plot = document.getElementById("{plot_id}");  // Note1 出力した<div>要素
var info = document.getElementById("plotly-node-info"); // Note2 tableを表示するための<div>要素（今回は、plotly-node-infoというID）

plot.on('plotly_selected', function (data) { // Note3
    {
        console.log('test');
        console.log(data);
        console.log(data.points);
        console.log('check');
        console.log(data.points[0].x); // これが日付
        data.points[0].data.marker.color = 'rgba(197,76,130,0.9)';
        console.log('???');
        var points = data.points;

        // クリックしたら色が変わる様にしたいな-
        // if(points.data.marker.color=='#6495ED'){
        //     points.data.marker.color = '#C54C82';
        // }
        // data.points[i].data.marker.color;
        // while (info.firstChild) info.removeChild(info.firstChild);
        // for (p in points) {
        //     info.appendChild(DictToTable(points[p].customdata));  // Note4
        // }
    }
})

// function DictToTable(data) {
//     var table = document.createElement("table");
//     for (key in data) {
//         var tr = document.createElement('tr');
//         var th = document.createElement('th');
//         var td = document.createElement('td');
//         th.textContent = key;
//         td.textContent = data[key];
//         tr.appendChild(th);
//         tr.appendChild(td);
//         table.appendChild(tr);
//     }
//     return table;
// }