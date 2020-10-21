/**
 * Polyfill for "fixing" IE's lack of support (IE < 9) for applying slice
 * on host objects like NamedNodeMap, NodeList, and HTMLCollection
 * (technically, since host objects are implementation-dependent,
 * IE hasn't needed to work this way, though I am informed this
 * will be changing with ES6). Also works on strings,
 * fixes IE to allow an explicit undefined for the 2nd argument
 * (as in Firefox), and prevents errors when called on other
 * DOM objects.
 * @license MIT, GPL, do whatever you want
-* @see https://gist.github.com/brettz9/6093105
*/
(function () {
    'use strict';
    var _slice = Array.prototype.slice;

    try {
        _slice.call(document.documentElement); // Can't be used with DOM elements in IE < 9
    }
    catch (e) { // Fails in IE < 9
        Array.prototype.slice = function (begin, end) {
            var i, arrl = this.length, a = [];
            if (this.charAt) { // Although IE < 9 does not fail when applying Array.prototype.slice
                               // to strings, here we do have to duck-type to avoid failing
                               // with IE < 9's lack of support for string indexes
                for (i = 0; i < arrl; i++) {
                    a.push(this.charAt(i));
                }
            }
            else { // This will work for genuine arrays, array-like objects, NamedNodeMap (attributes, entities, notations), NodeList (e.g., getElementsByTagName), HTMLCollection (e.g., childNodes), and will not fail on other DOM objects (as do DOM elements in IE < 9)
                for (i = 0; i < this.length; i++) { // IE < 9 (at least IE < 9 mode in IE 10) does not work with node.attributes (NamedNodeMap) without a dynamically checked length here
                    a.push(this[i]);
                }
            }
            return _slice.call(a, begin, end || a.length); // IE < 9 gives errors here if end is allowed as undefined (as opposed to just missing) so we default ourselves
        };
    }
}());

/**
* @license MIT, GPL, do whatever you want
* @requires polyfill: Array.prototype.slice fix {@link https://gist.github.com/brettz9/6093105}
*/
if (!Array.from) {
    Array.from = function (object) {
        'use strict';
        return [].slice.call(object);
    };
}
