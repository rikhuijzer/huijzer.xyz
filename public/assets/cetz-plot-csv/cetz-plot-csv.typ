#import "@preview/cetz:0.3.2": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#set page(width: auto, height: auto, margin: 0.3cm)

#let data = csv("data.csv", row-type: dictionary)

#let widths = data.map(x => float(x.width))
#let heights = data.map(x => float(x.height))

#let apples = data.filter(x => x.fruit == "apple").map(x => (float(x.width), float(x.height)))
#let pears = data.filter(x => x.fruit == "pear").map(x => (float(x.width), float(x.height)))

// To turn the plot into a scatter plot.
#let style = (stroke: none)

#let space = 0.3

// The first two colors from the Wong Color Palette.
#let wong-blue = rgb(0, 114, 178)
#let wong-orange = rgb(230, 159, 0)

#canvas({
  import draw: *
  
  plot.plot(
    size: (8, 8),
    axis-style: none,
    legend: "inner-north-west",
    {
      plot.add(
        pears,
        mark: "o",
        label: "Pear",
        style: style,
        mark-style: (fill: wong-blue)
      )
      plot.add(
        apples,
        mark: "x",
        label: "Apple",
        style: style,
        mark-style: (stroke: wong-orange)
      )
    }
  )
})
