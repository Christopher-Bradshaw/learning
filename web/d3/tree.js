var dataRaw = {
    name: "parent",
    children: [
        {
            name: "child1",
            children: [
                {name: "grandchild1"}
        ]},
        {name: "child2"},
    ]
};



var data = d3.hierarchy(dataRaw, function (d) {return d.children});

var treeChart = d3.tree();
// This is the default. Can do pretty cool things like using [360, rad] for a radial tree
// I don't quite understand this...
treeChart.size([1,1]);

var [width, height] = [500, 500];
var margin = 20;

function posX(x) {
    return x * (width - 2*margin) + margin;
}
function posY(y) {
    return y * (height - 2*margin) + margin;
}

// We now need to add features to our data that describe where it sits in the tree
// This adds x and y positions based on the treeChart.size
var data = treeChart(data);
console.log(data);

var fig = d3.select("svg");


// Create the nodes
fig.selectAll(".nodeG")
    .data(data)
    .enter()
    .append("g")
    .attr("class", "nodeG")
    .attr("transform", (d) => {
        return `translate(${posX(d.x)},${posY(d.y)})`}
    )
;

d3.selectAll(".nodeG")
    .append("circle")
    .attr("r", 10)
    .style("fill", "red")
;

// Create the links between those nodes
fig.append("g").attr("id", "treeG");

d3.select("#treeG")
    .selectAll("line")
    .data(data.descendants().slice(1))
    .enter()
    .insert("line")
    .attr("x1", d => posX(d.parent.x))
    .attr("y1", d => posY(d.parent.y))
    .attr("x2", d => posX(d.x))
    .attr("y2", d => posY(d.y))
    .style("stroke", "black")
;
