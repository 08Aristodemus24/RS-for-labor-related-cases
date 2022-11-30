import {createGlobalState} from "react-hooks-global-state";

const conf = {
    "automaticRearrangeAfterDropNode": false,
    "collapsible": false,
    "directed": false,
    "focusAnimationDuration": 0.75,
    "focusZoom": 3,
    "height": window.innerHeight - 50,
    "highlightDegree": 1,
    "highlightOpacity": 20,
    "linkHighlightBehavior": true,
    "maxZoom": 6,
    "minZoom": 0.1,
    "nodeHighlightBehavior": true,
    "panAndZoom": true,
    "staticGraph": false,
    "staticGraphWithDragAndDrop": false,
    "width": window.innerWidth - 10,
    "d3": {
        "alphaTarget": 0.05,
        "gravity": -200,
        "linkLength": 100,
        "linkStrength": 1
    },
    "node": {
        "color": "#0934a6",
        "fontColor": "black",
        "fontSize": 12,
        "fontWeight": "normal",
        "highlightColor": "#0936d4",
        "highlightFontSize": 12,
        "highlightFontWeight": "normal",
        "highlightStrokeColor": "SAME",
        "highlightStrokeWidth": "SAME",
        labelProperty: n => (n.name ? `${n.id} - ${n.name}` : n.id),
        "mouseCursor": "pointer",
        "opacity": 1,
        "renderLabel": true,
        "size": 400,
        "strokeColor": "none",
        "strokeWidth": 1.5,
        "svg": "",
        "symbolType": "circle"
    },
    "link": {
        "color": "#d3d3d3",
        "fontColor": "black",
        "fontSize": 12,
        "fontWeight": "normal",
        "highlightColor": "#508cf4",
        "highlightFontSize": 14,
        "highlightFontWeight": "normal",
        "labelProperty": "label",
        "mouseCursor": "pointer",
        "opacity": 1,
        "renderLabel": false,
        "semanticStrokeWidth": false,
        "strokeWidth": 1.5
    }
};

const initialState = {
    data: {
        links: [],
        nodes: [],
        duplicateMapping: [],
        mapping: {},
        groupedPersonAttributes: []
    },
    config: conf,
    availableItems: {
        subgraphs: [],
        attributes: {},
        models: {},
        graph: ""
    },
    options: {
        showDebuggingMode: true,
        default: {
            featureSet: undefined,
            subgraph: undefined
        },
        duplicateNodeHighlighting: {
            active: true
        },
        duplicateLinkHighlighting: {
            active: false,
            color: "firebrick",
            strokeWidth: {
                connected: 3,
                notConnected: 1
            }
        },
        attributeComparison: {
            showOnlyAvailableAttributes: true,
            showSubject: true,
            showMatching: true,
            matchingStyle: {
                fontWeight: "bold",
                color: "#1DC116"
            }
        },
        sideMenuResizeable: false,
        sideMenuWidth: 50,
        sideMenuOpacity: 0.75
    },
    subgraph: 0,
    selectedNodes: [],
    selectedFeatureSet: "n/a",
    selectedModels: [],
    predictions: []
};

export const {GlobalStateProvider, useGlobalState, setGlobalState} = createGlobalState(initialState);