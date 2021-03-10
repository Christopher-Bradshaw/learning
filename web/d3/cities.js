console.log("hi");

// Name, pop, happiness
var cities = [
    ["Sacramento", 300_000, 0.7],
    ["San Francisco", 700_000, 0.7],
    ["New York", 14_000_000, 0.6],
    ["London", 5_000_000, 0.8],
    ["Chicago", 6_000_000, 0.3],
];

const [x_max, y_max] = [500, 500];
const xscale = d3.scaleLinear()
    .domain([-0.5, cities.length - 1 + 0.5])
    .range([0, x_max]);

const yscale = d3.scaleLog()
    .domain([0.7 * d3.min(cities, el => el[1]), 1.3 * d3.max(cities, el => el[1])])
    .range([y_max, 0]);

const yaxis = d3.axisRight().scale(yscale).ticks(10).tickSize(10);

const happyscale = d3.scaleLinear().domain([0, 1]).range(["red", "blue"]);


d3.select("svg")
    .selectAll("circle")
    .data(cities)
    .enter()
    .append("circle")
    .attr("r", 20)
    .attr("cx", (d, i) => xscale(i))
    .attr("cy", d => (yscale(d[1]))) // 1 - y: as 0 is at the top
    .style("fill", d => happyscale(d[2]))
    .on("click", (_, d) => {
        console.log(d);
    });

d3.select("svg")
    .selectAll("text")
    .data(cities)
    .enter()
    .append("text")
    .attr("x", (d, i) => xscale(i))
    .attr("y", y_max)
    .text(d => d[0]);

d3.select("svg").append("g").attr("id", "yAxisG").call(yaxis)
d3.selectAll("#yAxisG").attr("transform","translate(0,0)")
