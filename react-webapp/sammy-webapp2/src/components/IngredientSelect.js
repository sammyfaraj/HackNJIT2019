import React from "react";
import WindowedSelect from "react-windowed-select";

export default class MultiSelect extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedList: []
    };
    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(selectedList) {
    this.setState({ selectedList });
    this.props.setSelected(selectedList);
  }


  render() {
    const { selectedList } = this.state;
    let type = this.props.type
    return (
      <WindowedSelect
        options={this.props.options[type]}
        onChange={this.handleChange}
        value={selectedList}
        isMulti
        closeMenuOnSelect={false}
      />
    );
  }
}


