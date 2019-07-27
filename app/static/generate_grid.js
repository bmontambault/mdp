String.prototype.formatUnicorn = String.prototype.formatUnicorn ||
function () {
    "use strict";
    var str = this.toString();
    if (arguments.length) {
        var t = typeof arguments[0];
        var key;
        var args = ("string" === t || "number" === t) ?
            Array.prototype.slice.call(arguments)
            : arguments[0];

        for (key in args) {
            str = str.replace(new RegExp("\\{" + key + "\\}", "gi"), args[key]);
        }
    }

    return str;
};

function generateGrid(rows, cols, colors) {

        var grid_num = 0
        var grid = "<table>";

        for ( row = 1; row <= rows; row++ ) {
            grid += "<tr>"; 
            for ( col = 1; col <= cols; col++ ) {      
                grid += "<td style='background-color:{color}'></td>".formatUnicorn({color:'#f00'});
                grid_num += 1
            }
            grid += "</tr>"; 
        }
        return grid;
    }